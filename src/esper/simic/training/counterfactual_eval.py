from __future__ import annotations

from typing import Any

from esper.leyline import SeedStage

from .parallel_env_state import ParallelEnvState


def reset_scaffolding_metrics(
    env_states: list[ParallelEnvState],
    slots: list[str],
) -> None:
    for env_state in env_states:
        for slot_id in slots:
            if env_state.model.has_active_seed_in_slot(slot_id):
                slot = env_state.model.seed_slots[slot_id]
                seed_state = slot.state
                if seed_state and seed_state.metrics:
                    seed_state.metrics.interaction_sum = 0.0
                    seed_state.metrics.boost_received = 0.0
                    seed_state.metrics.upstream_alpha_sum = 0.0
                    seed_state.metrics.downstream_alpha_sum = 0.0


def build_env_configs(
    env_states: list[ParallelEnvState],
    slots: list[str],
    epoch: int,
    max_epochs: int,
) -> list[list[dict[str, Any]]]:
    env_configs: list[list[dict[str, Any]]] = []
    for env_state in env_states:
        model = env_state.model
        active_slot_list = [
            sid
            for sid in slots
            if model.has_active_seed_in_slot(sid)
            and model.seed_slots[sid].state
            and model.seed_slots[sid].alpha > 0
        ]

        configs = [{"_kind": "main"}]

        if active_slot_list:
            for slot_id in active_slot_list:
                solo_config: dict[str, Any] = {
                    "_kind": "solo",
                    "_slot": slot_id,
                    slot_id: 0.0,
                }
                configs.append(solo_config)

            n_active = len(active_slot_list)
            if 2 <= n_active <= 4:
                all_off: dict[str, Any] = {sid: 0.0 for sid in active_slot_list}
                all_off["_kind"] = "all_off"
                configs.append(all_off)

            if 3 <= n_active <= 4:
                for idx_i in range(n_active):
                    for idx_j in range(idx_i + 1, n_active):
                        pair_config: dict[str, Any] = {
                            sid: 0.0
                            for k, sid in enumerate(active_slot_list)
                            if k != idx_i and k != idx_j
                        }
                        pair_config["_kind"] = "pair"
                        pair_config["_pair"] = (idx_i, idx_j)
                        configs.append(pair_config)

            committed_cfg: dict[str, Any] = {"_kind": "committed"}
            has_nonfossilized = False
            for slot_id in active_slot_list:
                slot_obj = model.seed_slots[slot_id]
                seed_state = slot_obj.state
                if seed_state is None:
                    continue
                if seed_state.stage != SeedStage.FOSSILIZED:
                    committed_cfg[slot_id] = 0.0
                    has_nonfossilized = True
            if has_nonfossilized:
                configs.append(committed_cfg)

        if (
            active_slot_list
            and env_state.counterfactual_helper
            and epoch == max_epochs
        ):
            required_configs = env_state.counterfactual_helper.get_required_configs(
                active_slot_list
            )
            for config_tuple in required_configs:
                shapley_cfg: dict[str, Any] = {
                    sid: 1.0 if enabled else 0.0
                    for sid, enabled in zip(active_slot_list, config_tuple)
                }
                shapley_cfg["_kind"] = "shapley"
                shapley_cfg["_tuple"] = config_tuple
                configs.append(shapley_cfg)

        env_configs.append(configs)

    return env_configs
