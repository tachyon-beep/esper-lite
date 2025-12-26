"""Policy factory for creating configured PolicyBundle instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from esper.leyline import DEFAULT_LSTM_HIDDEN_DIM

if TYPE_CHECKING:
    from esper.leyline.slot_config import SlotConfig
    from esper.tamiyo.policy.protocol import PolicyBundle

# Valid torch.compile modes
VALID_COMPILE_MODES = frozenset({"off", "default", "reduce-overhead", "max-autotune"})


def create_policy(
    policy_type: str = "lstm",
    state_dim: int | None = None,
    num_slots: int = 4,
    slot_config: "SlotConfig | None" = None,
    device: torch.device | str = "cpu",
    compile_mode: str = "off",
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
    **kwargs: Any,
) -> "PolicyBundle":
    """Create a policy instance with optional torch.compile.

    This is the recommended entry point for creating policies. It handles:
    - Policy instantiation via the registry
    - Device placement
    - torch.compile of the inner network (not the wrapper)

    Args:
        policy_type: Registered policy name (default: "lstm")
        state_dim: Observation feature dimension. If None, computed from slot_config.
        num_slots: Number of seed slots (deprecated - use slot_config instead)
        slot_config: Explicit slot configuration. If provided, num_slots is ignored.
            IMPORTANT: Always pass slot_config when using non-default slot layouts
            to ensure action heads and masks align correctly.
        device: Target device for the policy
        compile_mode: torch.compile mode ("off", "default", "reduce-overhead", "max-autotune")
        lstm_hidden_dim: Hidden dimension for LSTM policies
        **kwargs: Additional arguments passed to policy constructor

    Returns:
        Configured PolicyBundle instance on the target device

    Raises:
        ValueError: If compile_mode is not valid

    Example:
        >>> from esper.leyline.slot_config import SlotConfig
        >>> slot_config = SlotConfig.for_grid(rows=1, cols=4)
        >>> policy = create_policy(
        ...     policy_type="lstm",
        ...     slot_config=slot_config,
        ...     device="cuda:0",
        ...     compile_mode="default",
        ... )
    """
    # Local imports to avoid circular dependency:
    # factory.py imports from registry/features, which import from leyline,
    # which is imported at module level here. Keep these local for clean loading.
    from esper.tamiyo.policy.registry import get_policy
    from esper.tamiyo.policy.features import get_feature_size
    from esper.leyline.slot_config import SlotConfig

    # Validate compile_mode
    if compile_mode not in VALID_COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {sorted(VALID_COMPILE_MODES)}, got {compile_mode!r}"
        )

    # Use provided slot_config or create default from num_slots
    # IMPORTANT: Always prefer explicit slot_config for non-default layouts
    if slot_config is None:
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

    # Compile the policy via the protocol method
    # Note: Device placement MUST happen before compile for correct tracing
    # WARNING: Do not call .to(device) after this - it would replace the compiled module
    policy.compile(mode=compile_mode, dynamic=True)

    return policy


__all__ = ["create_policy"]
