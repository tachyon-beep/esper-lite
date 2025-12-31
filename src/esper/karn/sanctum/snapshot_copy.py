"""Copy helpers for Sanctum snapshots.

SanctumAggregator mutates internal state as telemetry events arrive.
The UI must never receive references to that live, mutable state because:

- A UI refresh can interleave with telemetry processing, yielding tearing.
- Accidental UI-side mutation would corrupt the aggregator.

These helpers build deep *structural* copies (new dataclass instances plus
new containers) without relying on generic deepcopy().
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import TypeVar

from esper.karn.sanctum.schema import (
    BestRunRecord,
    CounterfactualSnapshot,
    DecisionSnapshot,
    EnvState,
    EventLogEntry,
    GPUStats,
    RewardComponents,
    RunConfig,
    SeedState,
    SanctumSnapshot,
    SystemVitals,
    TamiyoState,
)

T = TypeVar("T")


def copy_deque(values: deque[T]) -> deque[T]:
    return deque(values, maxlen=values.maxlen)


def copy_reward_components(components: RewardComponents) -> RewardComponents:
    return replace(components)


def copy_counterfactual_snapshot(snapshot: CounterfactualSnapshot) -> CounterfactualSnapshot:
    return replace(
        snapshot,
        slot_ids=tuple(snapshot.slot_ids),
        configs=[
            replace(cfg, seed_mask=tuple(cfg.seed_mask))
            for cfg in snapshot.configs
        ],
    )


def copy_seed_state(seed: SeedState) -> SeedState:
    return replace(seed)


def copy_env_state(env: EnvState) -> EnvState:
    return replace(
        env,
        seeds={slot_id: copy_seed_state(seed) for slot_id, seed in env.seeds.items()},
        best_seeds={slot_id: copy_seed_state(seed) for slot_id, seed in env.best_seeds.items()},
        blueprint_spawns=dict(env.blueprint_spawns),
        blueprint_prunes=dict(env.blueprint_prunes),
        blueprint_fossilized=dict(env.blueprint_fossilized),
        reward_components=copy_reward_components(env.reward_components),
        counterfactual_matrix=copy_counterfactual_snapshot(env.counterfactual_matrix),
        reward_history=copy_deque(env.reward_history),
        accuracy_history=copy_deque(env.accuracy_history),
        action_history=copy_deque(env.action_history),
        action_counts=dict(env.action_counts),
    )


def copy_decision_snapshot(decision: DecisionSnapshot) -> DecisionSnapshot:
    return replace(
        decision,
        slot_states=dict(decision.slot_states),
        alternatives=list(decision.alternatives),
    )


def copy_run_config(config: RunConfig) -> RunConfig:
    return replace(config, entropy_anneal=dict(config.entropy_anneal))


def copy_tamiyo_state(tamiyo: TamiyoState) -> TamiyoState:
    return replace(
        tamiyo,
        episode_return_history=copy_deque(tamiyo.episode_return_history),
        policy_loss_history=copy_deque(tamiyo.policy_loss_history),
        value_loss_history=copy_deque(tamiyo.value_loss_history),
        grad_norm_history=copy_deque(tamiyo.grad_norm_history),
        entropy_history=copy_deque(tamiyo.entropy_history),
        explained_variance_history=copy_deque(tamiyo.explained_variance_history),
        kl_divergence_history=copy_deque(tamiyo.kl_divergence_history),
        clip_fraction_history=copy_deque(tamiyo.clip_fraction_history),
        action_counts=dict(tamiyo.action_counts),
        cumulative_action_counts=dict(tamiyo.cumulative_action_counts),
        recent_decisions=[copy_decision_snapshot(d) for d in tamiyo.recent_decisions],
        infrastructure=replace(tamiyo.infrastructure),
        gradient_quality=replace(tamiyo.gradient_quality),
        layer_gradient_health=(
            dict(tamiyo.layer_gradient_health) if tamiyo.layer_gradient_health is not None else None
        ),
    )


def copy_gpu_stats(stats: GPUStats) -> GPUStats:
    return replace(stats)


def copy_system_vitals(vitals: SystemVitals) -> SystemVitals:
    return replace(
        vitals,
        gpu_stats={device_id: copy_gpu_stats(stats) for device_id, stats in vitals.gpu_stats.items()},
    )


def copy_event_log_entry(entry: EventLogEntry) -> EventLogEntry:
    return replace(entry, metadata=dict(entry.metadata))


def copy_best_run_record(record: BestRunRecord) -> BestRunRecord:
    return replace(
        record,
        seeds={slot_id: copy_seed_state(seed) for slot_id, seed in record.seeds.items()},
        slot_ids=list(record.slot_ids),
        reward_components=(
            copy_reward_components(record.reward_components)
            if record.reward_components is not None
            else None
        ),
        counterfactual_matrix=(
            copy_counterfactual_snapshot(record.counterfactual_matrix)
            if record.counterfactual_matrix is not None
            else None
        ),
        action_history=list(record.action_history),
        reward_history=list(record.reward_history),
        accuracy_history=list(record.accuracy_history),
        blueprint_spawns=dict(record.blueprint_spawns),
        blueprint_fossilized=dict(record.blueprint_fossilized),
        blueprint_prunes=dict(record.blueprint_prunes),
    )


def copy_snapshot(snapshot: SanctumSnapshot) -> SanctumSnapshot:
    return replace(
        snapshot,
        envs={env_id: copy_env_state(env) for env_id, env in snapshot.envs.items()},
        tamiyo=copy_tamiyo_state(snapshot.tamiyo),
        vitals=copy_system_vitals(snapshot.vitals),
        rewards=copy_reward_components(snapshot.rewards),
        slot_ids=list(snapshot.slot_ids),
        run_config=copy_run_config(snapshot.run_config),
        mean_accuracy_history=copy_deque(snapshot.mean_accuracy_history),
        event_log=[copy_event_log_entry(e) for e in snapshot.event_log],
        best_runs=[copy_best_run_record(r) for r in snapshot.best_runs],
        cumulative_blueprint_spawns=dict(snapshot.cumulative_blueprint_spawns),
        cumulative_blueprint_fossilized=dict(snapshot.cumulative_blueprint_fossilized),
        cumulative_blueprint_prunes=dict(snapshot.cumulative_blueprint_prunes),
        slot_stage_counts=dict(snapshot.slot_stage_counts),
    )
