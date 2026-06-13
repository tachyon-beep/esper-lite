"""Lifecycle mutation contracts shared by Simic and Tolaria."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LifecycleMutationVerdict:
    """Tolaria pre-flight decision for a proposed lifecycle mutation."""

    approved: bool
    reason: str
