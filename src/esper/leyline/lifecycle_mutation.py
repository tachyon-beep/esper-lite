"""Lifecycle mutation contracts shared by Simic and Tolaria."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LifecycleMutationHealthSnapshot:
    """Structured Tolaria preflight inputs for lifecycle mutation decisions."""

    operation: str
    slot_id: str
    blueprint_id: str | None
    alpha_target: float | None
    alpha_speed_steps: int | None
    alpha_curve: str | None
    val_loss: float
    val_accuracy: float
    seed_stage: str | None
    total_params: int
    effective_seed_params: float
    max_seeds: int
    active_seed_count: int
    cooldown_epochs_remaining: int
    event_id: str


@dataclass(frozen=True, slots=True)
class LifecycleMutationVerdict:
    """Tolaria pre-flight decision for a proposed lifecycle mutation."""

    approved: bool
    reason: str
    blocked_factor: str | None = None
    health_snapshot: LifecycleMutationHealthSnapshot | None = None


@dataclass(frozen=True, slots=True)
class LifecycleMutationCausalContext:
    """Stable identity linking lifecycle proposal, verdict, mutation, and RNG lineage."""

    action_id: str
    proposal_id: str
    verdict_id: str
    mutation_id: str
    observation_hash: str
    rng_stream: str
    rng_seed: int
    topology: str
    slot_id: str
    operation: str
