"""Registry mapping LifecycleOp to handlers.

This module provides the dispatch registry that maps operation indices
to their handler functions. This enables the Strategy pattern for
lifecycle operations without modifying the main execution loop.

Usage:
    from esper.simic.training.handlers.registry import HANDLER_REGISTRY, CAN_EXECUTE_REGISTRY

    # Check if operation is valid
    can_fn = CAN_EXECUTE_REGISTRY.get(op_idx)
    if can_fn and can_fn(ctx):
        # Execute operation
        execute_fn = HANDLER_REGISTRY.get(op_idx)
        result = execute_fn(ctx, **params)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from esper.leyline import (
    OP_ADVANCE,
    OP_FOSSILIZE,
    OP_GERMINATE,
    OP_PRUNE,
    OP_SET_ALPHA_TARGET,
    OP_WAIT,
)
from esper.simic.training.handlers.advance import can_advance, execute_advance
from esper.simic.training.handlers.alpha import can_set_alpha_target, execute_set_alpha_target
from esper.simic.training.handlers.fossilize import can_fossilize, execute_fossilize
from esper.simic.training.handlers.germinate import can_germinate, execute_germinate
from esper.simic.training.handlers.prune import can_prune, execute_prune
from esper.simic.training.handlers.wait import can_wait, execute_wait

if TYPE_CHECKING:
    from esper.simic.training.handlers.base import HandlerContext, HandlerResult

# Type aliases for handler function signatures
CanExecuteFn = Callable[..., bool]
ExecuteFn = Callable[..., "HandlerResult"]

# Registry mapping operation indices to can_execute functions
# These functions check if an operation is valid in the current context
CAN_EXECUTE_REGISTRY: dict[int, CanExecuteFn] = {
    OP_WAIT: can_wait,
    OP_GERMINATE: can_germinate,
    OP_ADVANCE: can_advance,
    OP_FOSSILIZE: can_fossilize,
    OP_PRUNE: can_prune,
    OP_SET_ALPHA_TARGET: can_set_alpha_target,
}

# Registry mapping operation indices to execute functions
# These functions perform the actual operation
HANDLER_REGISTRY: dict[int, ExecuteFn] = {
    OP_WAIT: execute_wait,
    OP_GERMINATE: execute_germinate,
    OP_ADVANCE: execute_advance,
    OP_FOSSILIZE: execute_fossilize,
    OP_PRUNE: execute_prune,
    OP_SET_ALPHA_TARGET: execute_set_alpha_target,
}


def get_handler(op_idx: int) -> ExecuteFn | None:
    """Get the handler function for a lifecycle operation.

    Args:
        op_idx: Operation index (OP_WAIT, OP_GERMINATE, etc.)

    Returns:
        Handler function or None if operation not registered.
    """
    return HANDLER_REGISTRY.get(op_idx)


def get_can_execute(op_idx: int) -> CanExecuteFn | None:
    """Get the precondition check function for a lifecycle operation.

    Args:
        op_idx: Operation index (OP_WAIT, OP_GERMINATE, etc.)

    Returns:
        Precondition function or None if operation not registered.
    """
    return CAN_EXECUTE_REGISTRY.get(op_idx)


__all__ = [
    "CAN_EXECUTE_REGISTRY",
    "CanExecuteFn",
    "ExecuteFn",
    "HANDLER_REGISTRY",
    "get_can_execute",
    "get_handler",
]
