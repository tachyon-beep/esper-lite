"""Interactive Sanctum preview with staged data — no training run required.

Run from a terminal (>= 140x50 recommended, 120x40 minimum):

    uv run python -m esper.karn.sanctum.preview

Feeds the app a staged two-leg A/B run (leg A: HRA + RVT + PHN on; leg B:
control with RVT only) with a light random walk per poll so sparklines and
trends move. This is a LAYOUT preview harness for UI work: the numbers are
plausible, not real. For real data, launch training with ``--sanctum``.

Keys to try: [ / ] cycle Overview/Policy/Critic/Experiment, t switches the
primary leg, i run info, ? help + glossary, q quit.
"""

from __future__ import annotations

import random

from esper.karn.sanctum.app import SanctumApp
from esper.karn.sanctum.schema import EnvState, SanctumSnapshot
from esper.karn.sanctum.widgets.reward_health import RewardHealthData


def _build_leg(
    rng: random.Random,
    group: str,
    *,
    hra: bool,
    phn: bool,
    ev: float,
    ev_main: float | None,
    cf_share: float,
    acc: float,
) -> SanctumSnapshot:
    s = SanctumSnapshot(
        connected=True,
        task_name="cifar10_ab_preview",
        current_episode=47,
        current_epoch=150,
        max_epochs=500,
        current_batch=25,
        max_batches=100,
        runtime_seconds=4980.0,
        staleness_seconds=0.4,
        aggregate_mean_accuracy=acc,
        aggregate_mean_reward=11.9,
        reward_mode="shaped",
    )
    s.run_config.proof_profile = "ev-stab-s2"
    s.training_thread_alive = True

    t = s.tamiyo
    t.group_id = group
    t.ppo_data_received = True
    t.entropy, t.kl_divergence, t.clip_fraction = 1.32, 0.014, 0.11
    t.explained_variance, t.value_nrmse = ev, 0.84
    t.policy_loss, t.value_loss, t.grad_norm = -0.021, 0.48, 2.1
    t.advantage_mean, t.advantage_std = 0.0, 1.0
    t.advantage_skewness, t.advantage_kurtosis = 0.2, 0.4
    t.advantage_positive_ratio = 0.48
    t.log_prob_min, t.log_prob_max = -14.2, -0.4
    t.value_mean, t.value_std, t.value_min, t.value_max = 3.2, 6.1, -8.4, 14.1
    t.min_sparse_head_advantage_std = 0.07
    for i in range(10):
        t.entropy_history.append(1.45 - 0.013 * i)
        t.explained_variance_history.append(ev - 0.10 + 0.011 * i)
        t.kl_divergence_history.append(0.02 - 0.0006 * i)
        t.clip_fraction_history.append(0.13 - 0.002 * i)
        t.policy_loss_history.append(-0.02 + rng.uniform(-0.004, 0.004))
        t.value_loss_history.append(0.55 - 0.007 * i)
        t.grad_norm_history.append(2.3 + rng.uniform(-0.2, 0.2))
        t.decision_density_history.append(0.84)
    t.episode_return_history.extend([9.8, 10.4, 11.1, 11.9, 12.4])

    if hra:
        t.hra_leg_active = True
        t.ev_main, t.ev_cf, t.cf_value_loss = ev_main, 0.11, 0.021
        for i in range(8):
            t.ev_main_history.append((ev_main or 0.3) - 0.08 + 0.012 * i)
            t.ev_cf_history.append(0.08 + 0.004 * i)
            t.cf_value_loss_history.append(0.028 - 0.001 * i)
    if phn:
        t.advantage_per_head_normalized = True
        t.advantage_norm_fellback_count = 2

    t.rvt_leg_active = True
    t.return_var_cf_share = cf_share
    t.return_var_main_share = round(0.99 - cf_share, 2)
    t.return_var_residual_share = 0.01
    for i in range(8):
        t.return_var_cf_share_history.append(cf_share - 0.04 + 0.006 * i)

    statuses = ["healthy", "healthy", "improving", "healthy", "stalled", "healthy"]
    for i in range(6):
        env = EnvState(env_id=i, status=statuses[i])
        env.host_accuracy = acc + rng.uniform(-3.0, 3.0)
        env.episode_return = 11.0 + rng.uniform(-2.0, 3.0)
        s.envs[i] = env
    s.mean_accuracy_history.extend(acc - 8 + 0.4 * j for j in range(20))
    return s


class PreviewBackend:
    """Staged-data stand-in for SanctumBackend (same two-method contract)."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._legs = {
            "A": _build_leg(
                self._rng, "A", hra=True, phn=True,
                ev=0.31, ev_main=0.42, cf_share=0.51, acc=71.2,
            ),
            "B": _build_leg(
                self._rng, "B", hra=False, phn=False,
                ev=0.18, ev_main=None, cf_share=0.53, acc=70.9,
            ),
        }
        # Showcase the triage spine: leg B has one env in a governor-rollback
        # state, so the strip leads with GOV ROLLBACK (B) and the Overview tab
        # carries a red severity badge — the headline of the design review.
        b_env = self._legs["B"].envs[0]
        b_env.rolled_back = True
        b_env.rollback_reason = "nan"

    def _drift(self, s: SanctumSnapshot) -> None:
        """Small random walk so trends/sparklines move between polls."""
        rng = self._rng
        t = s.tamiyo
        s.current_epoch = min(s.max_epochs, s.current_epoch + 1)
        s.runtime_seconds += 0.25
        t.entropy = max(0.4, t.entropy + rng.uniform(-0.01, 0.008))
        t.entropy_history.append(t.entropy)
        t.explained_variance += rng.uniform(-0.01, 0.012)
        t.explained_variance_history.append(t.explained_variance)
        t.kl_divergence = max(0.0, t.kl_divergence + rng.uniform(-0.002, 0.002))
        t.kl_divergence_history.append(t.kl_divergence)
        t.grad_norm = max(0.2, t.grad_norm + rng.uniform(-0.15, 0.15))
        t.grad_norm_history.append(t.grad_norm)
        if t.ev_main is not None:
            t.ev_main += rng.uniform(-0.008, 0.012)
            t.ev_main_history.append(t.ev_main)
        if t.return_var_cf_share is not None:
            t.return_var_cf_share += rng.uniform(-0.005, 0.005)
            t.return_var_cf_share_history.append(t.return_var_cf_share)
        for env in s.envs.values():
            env.host_accuracy = min(99.0, env.host_accuracy + rng.uniform(-0.1, 0.15))

    def get_all_snapshots(self) -> dict[str, SanctumSnapshot]:
        for leg in self._legs.values():
            self._drift(leg)
        return self._legs

    def compute_reward_health_by_group(self) -> dict[str, RewardHealthData]:
        return {
            "A": RewardHealthData(
                pbrs_fraction=0.23, anti_gaming_trigger_rate=0.012,
                ev_explained=self._legs["A"].tamiyo.explained_variance,
                value_nrmse=0.84, hypervolume=1.4,
            ),
            "B": RewardHealthData(
                pbrs_fraction=0.25, anti_gaming_trigger_rate=0.015,
                ev_explained=self._legs["B"].tamiyo.explained_variance,
                value_nrmse=0.97, hypervolume=1.2,
            ),
        }


def main() -> None:
    app = SanctumApp(backend=PreviewBackend(), num_envs=6, refresh_rate=4.0)
    app.run()


if __name__ == "__main__":
    main()
