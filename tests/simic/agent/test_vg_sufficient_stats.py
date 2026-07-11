"""EV-stab PDR-0060: per-update vg sufficient-statistic emission from PPO update().

The agent emits, once per update and on the raw scale, the means + Gram matrix
of the value/target vector family: (V, G) on the OFF leg, (V_main, V_cf,
G_main, G_cf) on the HRA ON leg. Each leg's key set is disjoint from the
other's so persisted payloads can never be cross-leg conflated.
"""

import math

from tests.simic.agent.test_q_finiteness_and_contract import _build_agent, _fill_buffer

OFF_LEG_KEYS = frozenset({
    "vg_count", "vg_mean_v", "vg_mean_g",
    "vg_gram_v_v", "vg_gram_v_g", "vg_gram_g_g",
})

ON_LEG_KEYS = frozenset({
    "vg_count",
    "vg_mean_v_main", "vg_mean_v_cf", "vg_mean_g_main", "vg_mean_g_cf",
    "vg_gram_v_main_v_main", "vg_gram_v_main_v_cf",
    "vg_gram_v_main_g_main", "vg_gram_v_main_g_cf",
    "vg_gram_v_cf_v_cf", "vg_gram_v_cf_g_main", "vg_gram_v_cf_g_cf",
    "vg_gram_g_main_g_main", "vg_gram_g_main_g_cf",
    "vg_gram_g_cf_g_cf",
})


def test_off_leg_update_emits_total_scale_vg_stats() -> None:
    """OFF leg: the matrix degenerates to (V, G) — total critic vs total returns."""
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"]
    for key in OFF_LEG_KEYS:
        assert key in metrics, f"missing OFF-leg vg stat '{key}'"
        assert math.isfinite(metrics[key]), f"non-finite vg stat '{key}'"
    for key in ON_LEG_KEYS - OFF_LEG_KEYS:
        assert key not in metrics, f"ON-leg vg stat '{key}' leaked onto the OFF leg"
    assert metrics["vg_count"] > 0


def test_on_leg_update_emits_per_stream_vg_stats() -> None:
    """ON leg: full 4-vector family (V_main, V_cf, G_main, G_cf) on the raw scale."""
    agent, slot_config = _build_agent(hra_value_decomposition=True)
    _fill_buffer(agent, slot_config)

    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"]
    for key in ON_LEG_KEYS:
        assert key in metrics, f"missing ON-leg vg stat '{key}'"
        assert math.isfinite(metrics[key]), f"non-finite vg stat '{key}'"
    for key in OFF_LEG_KEYS - ON_LEG_KEYS:
        assert key not in metrics, f"OFF-leg vg stat '{key}' leaked onto the ON leg"
    assert metrics["vg_count"] > 0


def _emit_payload_for(metrics: dict) -> object:
    """Drive the real reducer + emitter chain and return the captured payload.

    Closes the silent-failure hole around the emitter's leg-discriminating
    .get(): a wiring regression that drops a leg's vg keys would otherwise
    surface only as NULL view columns, never as an error.
    """
    from unittest.mock import MagicMock

    from esper.simic.telemetry.emitters import emit_ppo_update_event
    from esper.simic.training.vectorized import _aggregate_ppo_metrics

    aggregated = _aggregate_ppo_metrics([metrics])
    # Injected by the training loop (not update()) before emission; the emitter
    # reads them with direct access, so provide the loop's contract here.
    aggregated.setdefault("ppo_updates_count", 1)
    aggregated.setdefault("throughput_step_time_ms_sum", 1.0)
    aggregated.setdefault("throughput_dataloader_wait_ms_sum", 0.0)
    aggregated.setdefault("rollback_count", 0)
    aggregated.setdefault("rollback_steps_zeroed", 0)
    aggregated.setdefault("rollback_attempt_count", 0)
    aggregated.setdefault("rollback_unattributed_count", 0)

    hub = MagicMock()
    emit_ppo_update_event(
        hub=hub,
        metrics=aggregated,
        episodes_completed=1,
        batch_idx=0,
        epoch=1,
        optimizer=None,
        grad_norm=1.0,
        update_time_ms=1.0,
    )
    hub.emit.assert_called_once()
    return hub.emit.call_args[0][0].data


def test_off_leg_vg_stats_reach_payload_end_to_end() -> None:
    """Real OFF-leg update -> reducer -> emitter: all 6 OFF keys land non-None."""
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    payload = _emit_payload_for(agent.update(clear_buffer=True))

    for key in OFF_LEG_KEYS:
        assert getattr(payload, key) is not None, f"OFF-leg vg stat '{key}' lost in emission"
    for key in ON_LEG_KEYS - OFF_LEG_KEYS:
        assert getattr(payload, key) is None, f"ON-leg vg stat '{key}' set on the OFF leg"


def test_on_leg_vg_stats_reach_payload_end_to_end() -> None:
    """Real ON-leg update -> reducer -> emitter: all 15 ON keys land non-None."""
    agent, slot_config = _build_agent(hra_value_decomposition=True)
    _fill_buffer(agent, slot_config)

    payload = _emit_payload_for(agent.update(clear_buffer=True))

    for key in ON_LEG_KEYS:
        assert getattr(payload, key) is not None, f"ON-leg vg stat '{key}' lost in emission"
    for key in OFF_LEG_KEYS - ON_LEG_KEYS:
        assert getattr(payload, key) is None, f"OFF-leg vg stat '{key}' set on the ON leg"
