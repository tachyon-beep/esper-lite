"""Task B: K>1 restores the operative PPO trust-region guard (PDR-0077, Read B).

At K=1 (recurrent_n_epochs=1) the guard is STRUCTURALLY inert: the only epoch is the
anchor epoch, whose importance ratio is exactly 1.0 and whose diagnostic KL is
KL(theta||theta)=0 by construction, so the target-KL early-stop can never fire and the
ratio-clip never binds. K>1 adds post-anchor epochs where the parameters have moved, so
the ratio diverges from 1, approx_kl>0, and both the early-stop and the clip become
operative. These tests pin that behavioural difference so the default is not silently
reverted to 1.
"""

from __future__ import annotations

import pytest
import torch

from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy

from tests.simic.test_ppo_update_golden import _fill_buffer


def _run_update(recurrent_n_epochs: int, target_kl: float | None) -> dict:
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm", slot_config=slot_config, device="cpu", compile_mode="off"
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=target_kl,
        recurrent_n_epochs=recurrent_n_epochs,
    )
    _fill_buffer(agent, slot_config)
    return agent.update(clear_buffer=True)


def test_k1_trust_region_is_inert():
    """K=1: the diagnostic KL is 0 by construction and the early-stop can never fire."""
    metrics = _run_update(recurrent_n_epochs=1, target_kl=1e-8)
    assert float(metrics["approx_kl"]) == pytest.approx(0.0, abs=1e-7)
    # Even with a near-zero target_kl the guard cannot fire — there is no post-anchor epoch.
    assert metrics.get("early_stop_epoch") is None


def test_k4_trust_region_activates():
    """K=4 (the default): approx_kl > 0 on the post-anchor epochs — the guard is operative."""
    metrics = _run_update(recurrent_n_epochs=4, target_kl=None)
    assert float(metrics["approx_kl"]) > 0.0


def test_k4_early_stop_is_reachable_on_epochs_ge_1():
    """K=4 with a tight target_kl fires the early-stop, and it can only fire at epoch >= 1."""
    metrics = _run_update(recurrent_n_epochs=4, target_kl=1e-8)
    assert float(metrics["approx_kl"]) > 0.0
    early_stop_epoch = metrics.get("early_stop_epoch")
    assert early_stop_epoch is not None, "early-stop must be reachable at K>1"
    assert int(early_stop_epoch) >= 1, "early-stop cannot fire on the epoch-0 anchor (KL=0)"


def test_default_config_uses_k_gt_1():
    """The run-facing default must be K>1 so real runs get an operative trust region."""
    from esper.simic.training.config import TrainingConfig

    config = TrainingConfig()
    assert config.recurrent_n_epochs > 1
    assert config.recurrent_n_epochs == 4  # this codebase's empirically anchored value
    assert config.to_train_kwargs()["recurrent_n_epochs"] == 4
    # The agent's clip_value default (False) is required under K>1 (PPOAgent.__init__ raises on
    # clip_value=True with recurrent_n_epochs>1). The config does not force clip_value=True.
    assert "clip_value" not in config.to_ppo_kwargs()
