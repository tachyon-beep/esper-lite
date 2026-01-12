"""Lifecycle operation handlers using the Strategy pattern.

This package contains handlers for each lifecycle operation, extracted
from action_execution.py to improve testability and reduce cognitive load.

Each operation has:
- A can_X() function to check preconditions
- An execute_X() function to perform the operation
- Operation-specific parameter dataclasses where needed

The registry module provides dispatch mappings from operation indices
to handler functions.

Usage:
    from esper.simic.training.handlers import (
        HandlerContext,
        HandlerResult,
        HANDLER_REGISTRY,
        can_germinate,
        execute_germinate,
        GerminateParams,
    )

    # Build context
    ctx = HandlerContext(
        env_idx=0,
        slot_id="r0c0",
        env_state=env_state,
        model=model,
        slot=model.seed_slots["r0c0"],
        seed_state=None,
        epoch=5,
        max_epochs=150,
        episodes_completed=0,
    )

    # Check and execute
    if can_germinate(ctx):
        params = GerminateParams(blueprint_idx=1, style_idx=0, tempo_idx=1, alpha_target=1.0)
        result = execute_germinate(ctx, params)
        if result.success:
            print("Germination successful!")
"""

# Base types
from esper.simic.training.handlers.base import (
    HandlerContext,
    HandlerResult,
    LifecycleHandler,
)

# Individual handlers
from esper.simic.training.handlers.advance import (
    ADVANCEABLE_STAGES,
    can_advance,
    execute_advance,
)
from esper.simic.training.handlers.alpha import (
    ALPHA_TARGET_STAGES,
    AlphaTargetParams,
    can_set_alpha_target,
    execute_set_alpha_target,
)
from esper.simic.training.handlers.fossilize import (
    HindsightCreditResult,
    can_fossilize,
    clear_beneficiary_from_ledger,
    compute_hindsight_credit_for_beneficiary,
    execute_fossilize,
)
from esper.simic.training.handlers.germinate import (
    GerminateParams,
    can_germinate,
    execute_germinate,
)
from esper.simic.training.handlers.prune import (
    PRUNABLE_STAGES,
    PruneParams,
    can_prune,
    execute_prune,
)
from esper.simic.training.handlers.wait import (
    can_wait,
    execute_wait,
)

# Registry
from esper.simic.training.handlers.registry import (
    CAN_EXECUTE_REGISTRY,
    CanExecuteFn,
    ExecuteFn,
    HANDLER_REGISTRY,
    get_can_execute,
    get_handler,
)

__all__ = [
    # Base types
    "HandlerContext",
    "HandlerResult",
    "LifecycleHandler",
    # Germinate
    "GerminateParams",
    "can_germinate",
    "execute_germinate",
    # Advance
    "ADVANCEABLE_STAGES",
    "can_advance",
    "execute_advance",
    # Fossilize
    "HindsightCreditResult",
    "can_fossilize",
    "clear_beneficiary_from_ledger",
    "compute_hindsight_credit_for_beneficiary",
    "execute_fossilize",
    # Prune
    "PRUNABLE_STAGES",
    "PruneParams",
    "can_prune",
    "execute_prune",
    # Wait
    "can_wait",
    "execute_wait",
    # Alpha
    "ALPHA_TARGET_STAGES",
    "AlphaTargetParams",
    "can_set_alpha_target",
    "execute_set_alpha_target",
    # Registry
    "CAN_EXECUTE_REGISTRY",
    "CanExecuteFn",
    "ExecuteFn",
    "HANDLER_REGISTRY",
    "get_can_execute",
    "get_handler",
]
