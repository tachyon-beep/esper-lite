"""Kasmina seed management scaffolding.

Responsible for coordinating seed registration and applying Tamiyo commands.
Actual kernel grafting logic will land in Slice 1 (see backlog TKT-102).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from torch import nn

from esper.leyline import leyline_pb2

from .lifecycle import KasminaLifecycle, LifecycleEvent, LifecycleState


class BlueprintRuntime(Protocol):
    """Protocol for runtime kernel execution support (Tezzeret/Urza)."""

    def load_kernel(self, blueprint_id: str) -> nn.Module:
        """Load a compiled kernel module for grafting."""


@dataclass(slots=True)
class SeedContext:
    """Represents state tracked for each active seed."""

    seed_id: str
    lifecycle: KasminaLifecycle = field(default_factory=KasminaLifecycle)
    metadata: dict[str, str] = field(default_factory=dict)


class KasminaSeedManager:
    """Skeleton seed manager handling Tamiyo adaptation commands."""

    def __init__(self, runtime: BlueprintRuntime) -> None:
        self._runtime = runtime
        self._seeds: dict[str, SeedContext] = {}

    def handle_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Dispatch a Tamiyo command to the appropriate lifecycle handler."""

        if command.command_type == leyline_pb2.COMMAND_SEED and command.HasField("seed_operation"):
            raw_seed_id = (
                command.target_seed_id
                or command.seed_operation.parameters.get("seed_id", "")
            )
            seed_id = str(raw_seed_id)
            blueprint_id = command.seed_operation.blueprint_id
            operation = command.seed_operation.operation
            if operation == leyline_pb2.SEED_OP_GERMINATE:
                self._graft_seed(seed_id, blueprint_id)
            elif operation in (leyline_pb2.SEED_OP_CULL, leyline_pb2.SEED_OP_CANCEL):
                self._retire_seed(seed_id)
        elif command.command_type == leyline_pb2.COMMAND_OPTIMIZER:
            self._log_adjustment(command)
        else:
            self._noop(command)

    def seeds(self) -> dict[str, SeedContext]:
        """Return the tracked seed contexts."""

        return dict(self._seeds)

    def _graft_seed(self, seed_id: str, blueprint_id: str) -> None:
        if not seed_id:
            return

        context = self._seeds.setdefault(seed_id, SeedContext(seed_id))
        lifecycle = context.lifecycle
        if lifecycle.state == LifecycleState.DORMANT:
            lifecycle.apply(LifecycleEvent.REGISTER)

        if lifecycle.state == LifecycleState.REGISTERED:
            lifecycle.apply(LifecycleEvent.GERMINATE)
        elif lifecycle.state != LifecycleState.GERMINATING:
            # Already in an active state; skip re-germination
            pass
        kernel = self._runtime.load_kernel(blueprint_id)
        self._attach_kernel(seed_id, kernel)
        if lifecycle.state != LifecycleState.ACTIVE:
            lifecycle.apply(LifecycleEvent.ACTIVATE)

    def _retire_seed(self, seed_id: str) -> None:
        context = self._seeds.get(seed_id)
        if not context:
            return

        lifecycle = context.lifecycle
        if lifecycle.state in {LifecycleState.ACTIVE, LifecycleState.OBSERVING}:
            lifecycle.apply(LifecycleEvent.RETIRE)
            lifecycle.apply(LifecycleEvent.TERMINATE)
            self._seeds.pop(seed_id, None)

    def _attach_kernel(self, seed_id: str, kernel: nn.Module) -> None:
        """Placeholder for kernel attachment logic."""

        _ = (seed_id, kernel)

    def _log_adjustment(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command

    def _noop(self, command: leyline_pb2.AdaptationCommand) -> None:
        _ = command


__all__ = ["KasminaSeedManager", "BlueprintRuntime", "SeedContext"]
