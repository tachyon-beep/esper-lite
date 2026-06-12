"""Contract tests for PPO update metric type declarations."""

from esper.simic.agent.types import PPOUpdateMetrics


def test_ppo_update_metrics_declares_emitted_agent_keys():
    """PPOUpdateMetrics should list keys emitted by PPOAgent.update()."""
    annotations = PPOUpdateMetrics.__annotations__

    emitted_keys = {
        "advantage_kurtosis",
        "advantage_mean",
        "advantage_positive_ratio",
        "advantage_skewness",
        "advantage_std",
        "clip_fraction_negative",
        "clip_fraction_positive",
        "d5_pre_norm_advantage_std",
        "gradient_cv",
        "pre_norm_advantage_mean",
        "return_mean",
        "return_std",
        "value_max",
        "value_mean",
        "value_min",
        "value_std",
        "value_target_scale",
    }

    assert emitted_keys <= annotations.keys()


def test_ppo_update_metrics_excludes_stale_agent_keys():
    """PPOUpdateMetrics should not advertise keys the agent no longer emits."""
    annotations = PPOUpdateMetrics.__annotations__

    stale_keys = {
        "decision_density",
        "entropy_loss",
        "gradient_stats",
        "lstm_c_l2_total",
        "lstm_h_l2_total",
        "total_loss",
    }

    assert annotations.keys().isdisjoint(stale_keys)
